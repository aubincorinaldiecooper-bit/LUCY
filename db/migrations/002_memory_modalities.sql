-- Multimodal memory support (SimpleMem Omni): track what kind of media a memory
-- unit came from and where the media file lives. Text rows keep media_url NULL.

ALTER TABLE memory_units
  ADD COLUMN IF NOT EXISTS modality text NOT NULL DEFAULT 'text';

ALTER TABLE memory_units
  ADD COLUMN IF NOT EXISTS media_url text;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'memory_units_modality_check'
  ) THEN
    ALTER TABLE memory_units
      ADD CONSTRAINT memory_units_modality_check
      CHECK (modality IN ('text', 'audio', 'image', 'video'));
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memory_units_modality ON memory_units(modality);
