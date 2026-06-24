// Magic-link email content. Kept in one place so the copy/branding is easy to
// find and edit. Table-based layout with inline styles for broad email-client
// support (Gmail/Outlook/Apple Mail). Override the brand name with
// AUTH_EMAIL_BRAND if you want something other than "Elsewhere".
const BRAND = process.env.AUTH_EMAIL_BRAND?.trim() || "Elsewhere";

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function buildMagicLinkEmail(url: string): {
  subject: string;
  text: string;
  html: string;
} {
  const safeUrl = escapeHtml(url);
  const subject = `Your ${BRAND} sign-in link`;

  const text = [
    `Sign in to ${BRAND}`,
    "",
    "Tap the link below to sign in. It works once and expires in 5 minutes.",
    "",
    url,
    "",
    "If you didn't request this, you can safely ignore this email.",
  ].join("\n");

  const html = `<!doctype html>
<html lang="en">
  <body style="margin:0;padding:0;background:#0b0b0f;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0b0b0f;padding:32px 12px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
      <tr>
        <td align="center">
          <table role="presentation" align="center" width="100%" cellpadding="0" cellspacing="0" style="max-width:440px;margin:0 auto;background:#15151c;border-radius:16px;padding:36px 32px;">
            <tr><td style="color:#ffffff;font-size:20px;font-weight:600;padding-bottom:10px;">${escapeHtml(BRAND)}</td></tr>
            <tr><td style="color:#cfcfd6;font-size:15px;line-height:22px;padding-bottom:24px;">Tap the button below to sign in. This link works once and expires in 5 minutes.</td></tr>
            <tr>
              <td style="padding-bottom:24px;">
                <a href="${safeUrl}" style="display:inline-block;background:#ffffff;color:#000000;text-decoration:none;font-size:15px;font-weight:600;padding:12px 26px;border-radius:10px;">Sign in to ${escapeHtml(BRAND)}</a>
              </td>
            </tr>
            <tr><td style="color:#7c7c87;font-size:13px;line-height:20px;padding-bottom:24px;">If the button doesn't work, paste this link into your browser:<br /><a href="${safeUrl}" style="color:#9a9aff;word-break:break-all;">${safeUrl}</a></td></tr>
            <tr><td style="color:#5a5a63;font-size:12px;line-height:18px;border-top:1px solid #26262e;padding-top:20px;">You're receiving this because someone entered this email to sign in to ${escapeHtml(BRAND)}. If that wasn't you, you can safely ignore it.</td></tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>`;

  return { subject, text, html };
}
