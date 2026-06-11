// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI-Native Book + RAG Chatbot',
  tagline: 'Physical AI & Humanoid Robotics Documentation',
  favicon: 'img/favicon.svg',

  // Set the production url of your site here
  url: process.env.SITE_URL || 'https://ai-native-book.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'ai-native-book', // Usually your GitHub org/user name.
  projectName: 'ai-native-book', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Google Fonts for Aurora theme
  stylesheets: [
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap',
      type: 'text/css',
    },
  ],

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Custom fields for chatbot API configuration
  customFields: {
    chatbotApiUrl: process.env.CHATBOT_API_URL || 'https://kiran9naseer-ai-nativebook.hf.space',
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/ai-native-book/ai-native-book/tree/master/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/ai-native-book/ai-native-book/tree/master/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      // Default to dark mode for Aurora Borealis theme
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'AI-Native Book',
        logo: {
          alt: 'AI-Native Book Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          {
            href: 'https://github.com/ai-native-book/ai-native-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/ai-native-book/ai-native-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI-Native Book Project. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.dracula,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
};

module.exports = config;